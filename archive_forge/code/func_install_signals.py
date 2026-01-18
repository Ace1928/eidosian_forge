from . import _gi
def install_signals(cls):
    """Adds Signal instances on a GObject derived class into the '__gsignals__'
    dictionary to be picked up and registered as real GObject signals.
    """
    gsignals = cls.__dict__.get('__gsignals__', {})
    newsignals = {}
    for name, signal in cls.__dict__.items():
        if isinstance(signal, Signal):
            signalName = str(signal)
            if not signalName:
                signalName = name
                signal = signal.copy(name)
                setattr(cls, name, signal)
            if signalName in gsignals:
                raise ValueError('Signal "%s" has already been registered.' % name)
            newsignals[signalName] = signal
            gsignals[signalName] = signal.get_signal_args()
    cls.__gsignals__ = gsignals
    for name, signal in newsignals.items():
        if signal.func is not None:
            funcName = 'do_' + name.replace('-', '_')
            if not hasattr(cls, funcName):
                setattr(cls, funcName, signal.func)