def reactor_start(*args):
    """Start the twisted reactor main loop
        """
    Logger.info('Support: Starting twisted reactor')
    reactor.interleave(reactor_wake, **kwargs)
    Clock.schedule_interval(reactor_work, 0)