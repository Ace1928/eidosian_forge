import ctypes
from OpenGL import plugins
@classmethod
def loadPlugin(cls, entrypoint):
    """Load a single entry-point via plugins module"""
    if not entrypoint.loaded:
        from OpenGL.arrays.arraydatatype import ArrayDatatype
        try:
            plugin_class = entrypoint.load()
        except ImportError as err:
            from OpenGL import logs
            from OpenGL._configflags import WARN_ON_FORMAT_UNAVAILABLE
            _log = logs.getLog('OpenGL.formathandler')
            if WARN_ON_FORMAT_UNAVAILABLE:
                logFunc = _log.warn
            else:
                logFunc = _log.info
            logFunc('Unable to load registered array format handler %s:\n%s', entrypoint.name, _log.getException(err))
        else:
            handler = plugin_class()
            ArrayDatatype.getRegistry()[entrypoint.name] = handler
            return handler
        entrypoint.loaded = True