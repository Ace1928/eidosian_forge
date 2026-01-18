import sys
import sysconfig
def aix_buildtag():
    """
    Return the platform_tag of the system Python was built on.
    """
    build_date = sysconfig.get_config_var('AIX_BUILDDATE')
    try:
        build_date = int(build_date)
    except (ValueError, TypeError):
        raise ValueError(f'AIX_BUILDDATE is not defined or invalid: {build_date!r}')
    return _aix_tag(_aix_bgt(), build_date)