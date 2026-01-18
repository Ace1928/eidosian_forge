import os
def get_pkgconfig_dir() -> str:
    """
    Return the path to the pybind11 pkgconfig directory.
    """
    pkgconfig_installed_path = os.path.join(DIR, 'share', 'pkgconfig')
    if os.path.exists(pkgconfig_installed_path):
        return pkgconfig_installed_path
    msg = 'pybind11 not installed, installation required to access the pkgconfig files'
    raise ImportError(msg)