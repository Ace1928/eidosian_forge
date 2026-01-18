from ._memmapping_reducer import get_memmapping_reducers
from ._memmapping_reducer import TemporaryResourcesManager
from .externals.loky.reusable_executor import _ReusablePoolExecutor
@classmethod
def get_memmapping_executor(cls, n_jobs, timeout=300, initializer=None, initargs=(), env=None, temp_folder=None, context_id=None, **backend_args):
    """Factory for ReusableExecutor with automatic memmapping for large
        numpy arrays.
        """
    global _executor_args
    executor_args = backend_args.copy()
    executor_args.update(env if env else {})
    executor_args.update(dict(timeout=timeout, initializer=initializer, initargs=initargs))
    reuse = _executor_args is None or _executor_args == executor_args
    _executor_args = executor_args
    manager = TemporaryResourcesManager(temp_folder)
    job_reducers, result_reducers = get_memmapping_reducers(unlink_on_gc_collect=True, temp_folder_resolver=manager.resolve_temp_folder_name, **backend_args)
    _executor, executor_is_reused = super().get_reusable_executor(n_jobs, job_reducers=job_reducers, result_reducers=result_reducers, reuse=reuse, timeout=timeout, initializer=initializer, initargs=initargs, env=env)
    if not executor_is_reused:
        _executor._temp_folder_manager = manager
    if context_id is not None:
        _executor._temp_folder_manager.register_new_context(context_id)
    return _executor