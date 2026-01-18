import os
import pathlib
import shutil
import nox
@nox.session(python='3.8')
def mypy(session):
    """Verify type hints are mypy compatible."""
    session.install('-e', '.')
    session.install('mypy', 'types-cachetools', 'types-certifi', 'types-freezegun', 'types-pyOpenSSL', 'types-requests', 'types-setuptools', 'types-six', 'types-mock')
    session.run('mypy', 'google/', 'tests/', 'tests_async/')