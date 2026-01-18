def install_twisted_reactor(**kwargs):
    """Installs a threaded twisted reactor, which will schedule one
    reactor iteration before the next frame only when twisted needs
    to do some work.

    Any arguments or keyword arguments passed to this function will be
    passed on the threadedselect reactors interleave function. These
    are the arguments one would usually pass to twisted's reactor.startRunning.

    Unlike the default twisted reactor, the installed reactor will not handle
    any signals unless you set the 'installSignalHandlers' keyword argument
    to 1 explicitly. This is done to allow kivy to handle the signals as
    usual unless you specifically want the twisted reactor to handle the
    signals (e.g. SIGINT).

    .. note::
        Twisted is not included in iOS build by default. To use it on iOS,
        put the twisted distribution (and zope.interface dependency) in your
        application directory.
    """
    import twisted
    if hasattr(twisted, '_kivy_twisted_reactor_installed'):
        return
    twisted._kivy_twisted_reactor_installed = True
    kwargs.setdefault('installSignalHandlers', 0)
    from twisted.internet import _threadedselect
    _threadedselect.install()
    from twisted.internet import reactor
    from twisted.internet.error import ReactorNotRunning
    from collections import deque
    from kivy.base import EventLoop
    from kivy.logger import Logger
    from kivy.clock import Clock
    q = deque()

    def reactor_wake(twisted_loop_next):
        """Wakeup the twisted reactor to start processing the task queue
        """
        Logger.trace('Support: twisted wakeup call to schedule task')
        q.append(twisted_loop_next)

    def reactor_work(*args):
        """Process the twisted reactor task queue
        """
        Logger.trace('Support: processing twisted task queue')
        while len(q):
            q.popleft()()
    global _twisted_reactor_work
    _twisted_reactor_work = reactor_work

    def reactor_start(*args):
        """Start the twisted reactor main loop
        """
        Logger.info('Support: Starting twisted reactor')
        reactor.interleave(reactor_wake, **kwargs)
        Clock.schedule_interval(reactor_work, 0)

    def reactor_stop(*args):
        """Shutdown the twisted reactor main loop
        """
        if reactor.threadpool:
            Logger.info('Support: Stopping twisted threads')
            reactor.threadpool.stop()
        Logger.info('Support: Shutting down twisted reactor')
        reactor._mainLoopShutdown()
        try:
            reactor.stop()
        except ReactorNotRunning:
            pass
        import sys
        sys.modules.pop('twisted.internet.reactor', None)
    global _twisted_reactor_stopper
    _twisted_reactor_stopper = reactor_stop
    Clock.schedule_once(reactor_start, 0)
    EventLoop.bind(on_stop=reactor_stop)