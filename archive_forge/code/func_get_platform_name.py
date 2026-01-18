import sys
def get_platform_name():
    if sys.platform.startswith('win'):
        return PLATFORM_WINDOWS
    elif sys.platform.startswith('darwin'):
        return PLATFORM_DARWIN
    elif sys.platform.startswith('linux'):
        return PLATFORM_LINUX
    elif sys.platform.startswith(('dragonfly', 'freebsd', 'netbsd', 'openbsd')):
        return PLATFORM_BSD
    else:
        return PLATFORM_UNKNOWN