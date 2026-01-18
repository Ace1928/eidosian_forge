import json
from pathlib import Path
import pytest
from jupyter_server.services.contents.filemanager import AsyncFileContentsManager
from jupyter_server.services.contents.largefilemanager import AsyncLargeFileManager
@pytest.fixture()
def jp_kernelspecs(jp_data_dir: Path) -> None:
    """Configures some sample kernelspecs in the Jupyter data directory."""
    spec_names = ['sample', 'sample2', 'bad']
    for name in spec_names:
        sample_kernel_dir = jp_data_dir.joinpath('kernels', name)
        sample_kernel_dir.mkdir(parents=True)
        sample_kernel_file = sample_kernel_dir.joinpath('kernel.json')
        kernel_json = sample_kernel_json.copy()
        if name == 'bad':
            kernel_json['argv'] = ['non_existent_path']
        sample_kernel_file.write_text(json.dumps(kernel_json))
        sample_kernel_resources = sample_kernel_dir.joinpath('resource.txt')
        sample_kernel_resources.write_text(some_resource)