import os
import zipfile
import logging
import tempfile
import uuid
import shutil
from ..utils import download, check_sha1, replace_file
from ... import base
def get_model_file(name, root=os.path.join(base.data_dir(), 'models')):
    """Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name + '.params')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning('Mismatch in the content of model file detected. Downloading again.')
    else:
        logging.info('Model file not found. Downloading to %s.', file_path)
    os.makedirs(root, exist_ok=True)
    repo_url = os.environ.get('MXNET_GLUON_REPO', apache_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    random_uuid = str(uuid.uuid4())
    temp_zip_file_path = os.path.join(root, file_name + '.zip' + random_uuid)
    download(_url_format.format(repo_url=repo_url, file_name=file_name), path=temp_zip_file_path, overwrite=True)
    with zipfile.ZipFile(temp_zip_file_path) as zf:
        temp_dir = tempfile.mkdtemp(dir=root)
        zf.extractall(temp_dir)
        temp_file_path = os.path.join(temp_dir, file_name + '.params')
        replace_file(temp_file_path, file_path)
        shutil.rmtree(temp_dir)
    os.remove(temp_zip_file_path)
    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')