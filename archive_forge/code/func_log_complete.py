from . import logger
import logging
def log_complete(task, logger='TaskLogger'):
    """Complete logging of a task

    Convenience function to log a task in the default
    TaskLogger

    Parameters
    ----------
    task : str
        Name of the task to be started
    logger : str, optional (default: "TaskLogger")
        Unique name of the logger to retrieve

    Returns
    -------
    time : float
        The time lapsed between task start and completion
    """
    tasklogger = get_tasklogger(logger)
    return tasklogger.complete_task(task)