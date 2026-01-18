import os
import platform
import re
from pathlib import Path
import pytest
import sklearn
from sklearn._min_dependencies import dependent_packages
from sklearn.utils.fixes import parse_version
def test_min_dependencies_readme():
    if platform.python_implementation() == 'PyPy':
        pytest.skip('PyPy does not always share the same minimum deps')
    pattern = re.compile('(\\.\\. \\|)' + '(([A-Za-z]+\\-?)+)' + '(MinVersion\\| replace::)' + '( [0-9]+\\.[0-9]+(\\.[0-9]+)?)')
    readme_path = Path(sklearn.__file__).parent.parent
    readme_file = readme_path / 'README.rst'
    if not os.path.exists(readme_file):
        pytest.skip('The README.rst file is not available.')
    with readme_file.open('r') as f:
        for line in f:
            matched = pattern.match(line)
            if not matched:
                continue
            package, version = (matched.group(2), matched.group(5))
            package = package.lower()
            if package in dependent_packages:
                version = parse_version(version)
                min_version = parse_version(dependent_packages[package][0])
                assert version == min_version, f'{package} has a mismatched version'