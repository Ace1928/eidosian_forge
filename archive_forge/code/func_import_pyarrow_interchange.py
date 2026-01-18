from types import ModuleType
from packaging.version import Version
from importlib.metadata import version as importlib_version
def import_pyarrow_interchange() -> ModuleType:
    min_version = '11.0.0'
    try:
        version = importlib_version('pyarrow')
        if Version(version) < Version(min_version):
            raise RuntimeError(f'The pyarrow package must be version {min_version} or greater. Found version {version}')
        import pyarrow.interchange as pi
        return pi
    except ImportError as err:
        raise ImportError(f'Usage of the DataFrame Interchange Protocol requires\nversion {min_version} or greater of the pyarrow package. \nThis can be installed with pip using:\n   pip install "pyarrow>={min_version}"\nor conda:\n   conda install -c conda-forge "pyarrow>={min_version}"\n\nImportError: {err.args[0]}') from err