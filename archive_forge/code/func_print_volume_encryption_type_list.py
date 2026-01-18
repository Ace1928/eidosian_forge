import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def print_volume_encryption_type_list(encryption_types):
    """
    Lists volume encryption types.

    :param encryption_types: a list of :class: VolumeEncryptionType instances
    """
    print_list(encryption_types, ['Volume Type ID', 'Provider', 'Cipher', 'Key Size', 'Control Location'])