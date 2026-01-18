from types import ModuleType
from packaging.version import Version
from importlib.metadata import version as importlib_version
def import_vegafusion() -> ModuleType:
    min_version = '1.5.0'
    try:
        version = importlib_version('vegafusion')
        embed_version = importlib_version('vegafusion-python-embed')
        if version != embed_version or Version(version) < Version(min_version):
            raise RuntimeError(f'The versions of the vegafusion and vegafusion-python-embed packages must match\nand must be version {min_version} or greater.\nFound:\n - vegafusion=={version}\n - vegafusion-python-embed=={embed_version}\n')
        import vegafusion as vf
        return vf
    except ImportError as err:
        raise ImportError(f"""The "vegafusion" data transformer and chart.transformed_data feature requires\nversion {min_version} or greater of the 'vegafusion-python-embed' and 'vegafusion' packages.\nThese can be installed with pip using:\n    pip install "vegafusion[embed]>={min_version}"\nOr with conda using:\n    conda install -c conda-forge "vegafusion-python-embed>={min_version}" "vegafusion>={min_version}"\n\nImportError: {err.args[0]}""") from err