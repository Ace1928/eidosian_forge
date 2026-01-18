import subprocess
import sys
import sys
import geopandas
def test_no_additional_imports():
    blacklist = {'pytest', 'py', 'ipython', 'mapclassify', 'sqlalchemy', 'psycopg2', 'geopy', 'geoalchemy2', 'matplotlib'}
    code = "\nimport sys\nimport geopandas\nblacklist = {0!r}\n\nmods = blacklist & set(m.split('.')[0] for m in sys.modules)\nif mods:\n    sys.stderr.write('err: geopandas should not import: {{}}'.format(', '.join(mods)))\n    sys.exit(len(mods))\n".format(blacklist)
    call = [sys.executable, '-c', code]
    returncode = subprocess.run(call).returncode
    assert returncode == 0