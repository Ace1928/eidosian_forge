def reactor_wake(twisted_loop_next):
    """Wakeup the twisted reactor to start processing the task queue
        """
    Logger.trace('Support: twisted wakeup call to schedule task')
    q.append(twisted_loop_next)