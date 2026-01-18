import inspect
import logging
import sys
def set_ev_cls(ev_cls, dispatchers=None):
    """
    A decorator for OSKen application to declare an event handler.

    Decorated method will become an event handler.
    ev_cls is an event class whose instances this OSKenApp wants to receive.
    dispatchers argument specifies one of the following negotiation phases
    (or a list of them) for which events should be generated for this handler.
    Note that, in case an event changes the phase, the phase before the change
    is used to check the interest.

    .. tabularcolumns:: |l|L|

    ============================================== ===============================
    Negotiation phase                              Description
    ============================================== ===============================
    os_ken.controller.handler.HANDSHAKE_DISPATCHER Sending and waiting for hello
                                                   message
    os_ken.controller.handler.CONFIG_DISPATCHER    Version negotiated and sent
                                                   features-request message
    os_ken.controller.handler.MAIN_DISPATCHER      Switch-features message
                                                   received and sent set-config
                                                   message
    os_ken.controller.handler.DEAD_DISPATCHER      Disconnect from the peer.  Or
                                                   disconnecting due to some
                                                   unrecoverable errors.
    ============================================== ===============================
    """

    def _set_ev_cls_dec(handler):
        if 'callers' not in dir(handler):
            handler.callers = {}
        for e in _listify(ev_cls):
            handler.callers[e] = _Caller(_listify(dispatchers), e.__module__)
        return handler
    return _set_ev_cls_dec