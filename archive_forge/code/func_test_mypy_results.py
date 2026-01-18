import os
import re
import shutil
import sys
from pathlib import Path
import pytest
@pytest.mark.skipif(mypy.__file__.endswith('.py'), reason='Non-compiled mypy is too slow')
@pytest.mark.parametrize('config_filename,python_filename,output_filename', cases)
def test_mypy_results(config_filename, python_filename, output_filename, tmpdir, monkeypatch):
    from mypy import api as mypy_api
    os.chdir(tmpdir)
    root_dir = Path(__file__).parent
    thinc_root_dir = Path(__file__).parent.parent.parent.parent
    if '--pyargs' not in sys.argv:
        monkeypatch.setenv('MYPYPATH', str(thinc_root_dir))
    tmpdir_path = Path(tmpdir)
    full_config_path: Path = root_dir / f'configs/{config_filename}'
    full_module_path: Path = root_dir / f'modules/{python_filename}'
    full_output_path: Path = root_dir / f'outputs/{output_filename}'
    full_tmp_config_path: Path = tmpdir_path / config_filename
    full_tmp_module_path: Path = tmpdir_path / python_filename
    shutil.copy(str(full_config_path), tmpdir)
    shutil.copy(str(full_module_path), tmpdir)
    expected_out = ''
    expected_err = ''
    expected_returncode = 1
    expected_out = full_output_path.read_text()
    cache_dir = tmpdir_path / f'.mypy_cache/test-{config_filename[:-4]}'
    command = [str(full_tmp_module_path), '--config-file', str(full_tmp_config_path), '--cache-dir', str(cache_dir), '--show-error-codes']
    print(f'\nExecuting: mypy {' '.join(command)}')
    actual_result = mypy_api.run(command)
    actual_out, actual_err, actual_returncode = actual_result
    actual_out = '\n'.join(['.py:'.join(line.split('.py:')[1:]) for line in actual_out.split('\n') if line]).strip()
    actual_out = re.sub('\\n\\s*\\n', '\\n', actual_out)
    if GENERATE and output_filename is not None:
        full_output_path.write_text(actual_out)
    else:
        assert actual_out.strip() == expected_out.strip(), actual_out
    assert actual_err == expected_err
    assert actual_returncode == expected_returncode