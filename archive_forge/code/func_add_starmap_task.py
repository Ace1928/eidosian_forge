from celery._state import connect_on_app_finalize
from celery.utils.log import get_logger
@connect_on_app_finalize
def add_starmap_task(app):
    from celery.canvas import signature

    @app.task(name='celery.starmap', shared=False, lazy=False)
    def xstarmap(task, it):
        task = signature(task, app=app).type
        return [task(*item) for item in it]
    return xstarmap