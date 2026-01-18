import io
import json
import six
from google.auth import crypt
from google.auth import exceptions
def from_filename(filename, require=None, use_rsa_signer=True):
    """Reads a Google service account JSON file and returns its parsed info.

    Args:
        filename (str): The path to the service account .json file.
        require (Sequence[str]): List of keys required to be present in the
            info.
        use_rsa_signer (Optional[bool]): Whether to use RSA signer or EC signer.
            We use RSA signer by default.

    Returns:
        Tuple[ Mapping[str, str], google.auth.crypt.Signer ]: The verified
            info and a signer instance.
    """
    with io.open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        return (data, from_dict(data, require=require, use_rsa_signer=use_rsa_signer))