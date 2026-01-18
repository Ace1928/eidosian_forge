import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
def generate_sub_moved_events(src_dir_path, dest_dir_path):
    """Generates an event list of :class:`DirMovedEvent` and
    :class:`FileMovedEvent` objects for all the files and directories within
    the given moved directory that were moved along with the directory.

    :param src_dir_path:
        The source path of the moved directory.
    :param dest_dir_path:
        The destination path of the moved directory.
    :returns:
        An iterable of file system events of type :class:`DirMovedEvent` and
        :class:`FileMovedEvent`.
    """
    for root, directories, filenames in os.walk(dest_dir_path):
        for directory in directories:
            full_path = os.path.join(root, directory)
            renamed_path = full_path.replace(dest_dir_path, src_dir_path) if src_dir_path else None
            yield DirMovedEvent(renamed_path, full_path)
        for filename in filenames:
            full_path = os.path.join(root, filename)
            renamed_path = full_path.replace(dest_dir_path, src_dir_path) if src_dir_path else None
            yield FileMovedEvent(renamed_path, full_path)