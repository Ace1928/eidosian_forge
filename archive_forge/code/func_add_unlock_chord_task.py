from celery._state import connect_on_app_finalize
from celery.utils.log import get_logger
@connect_on_app_finalize
def add_unlock_chord_task(app):
    """Task used by result backends without native chord support.

    Will joins chord by creating a task chain polling the header
    for completion.
    """
    from celery.canvas import maybe_signature
    from celery.exceptions import ChordError
    from celery.result import allow_join_result, result_from_tuple

    @app.task(name='celery.chord_unlock', max_retries=None, shared=False, default_retry_delay=app.conf.result_chord_retry_interval, ignore_result=True, lazy=False, bind=True)
    def unlock_chord(self, group_id, callback, interval=None, max_retries=None, result=None, Result=app.AsyncResult, GroupResult=app.GroupResult, result_from_tuple=result_from_tuple, **kwargs):
        if interval is None:
            interval = self.default_retry_delay
        callback = maybe_signature(callback, app)
        deps = GroupResult(group_id, [result_from_tuple(r, app=app) for r in result], app=app)
        j = deps.join_native if deps.supports_native_join else deps.join
        try:
            ready = deps.ready()
        except Exception as exc:
            raise self.retry(exc=exc, countdown=interval, max_retries=max_retries)
        else:
            if not ready:
                raise self.retry(countdown=interval, max_retries=max_retries)
        callback = maybe_signature(callback, app=app)
        try:
            with allow_join_result():
                ret = j(timeout=app.conf.result_chord_join_timeout, propagate=True)
        except Exception as exc:
            try:
                culprit = next(deps._failed_join_report())
                reason = f'Dependency {culprit.id} raised {exc!r}'
            except StopIteration:
                reason = repr(exc)
            logger.exception('Chord %r raised: %r', group_id, exc)
            app.backend.chord_error_from_stack(callback, ChordError(reason))
        else:
            try:
                callback.delay(ret)
            except Exception as exc:
                logger.exception('Chord %r raised: %r', group_id, exc)
                app.backend.chord_error_from_stack(callback, exc=ChordError(f'Callback error: {exc!r}'))
    return unlock_chord