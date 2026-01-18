from __future__ import annotations
from collections import abc
from typing import Any, Mapping, Optional, cast
from bson.objectid import ObjectId
from gridfs.errors import NoFile
from gridfs.grid_file import (
from pymongo import ASCENDING, DESCENDING, _csot
from pymongo.client_session import ClientSession
from pymongo.collection import Collection
from pymongo.common import validate_string
from pymongo.database import Database
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import _ServerMode
from pymongo.write_concern import WriteConcern
def open_download_stream(self, file_id: Any, session: Optional[ClientSession]=None) -> GridOut:
    """Opens a Stream from which the application can read the contents of
        the stored file specified by file_id.

        For example::

          my_db = MongoClient().test
          fs = GridFSBucket(my_db)
          # get _id of file to read.
          file_id = fs.upload_from_stream("test_file", "data I want to store!")
          grid_out = fs.open_download_stream(file_id)
          contents = grid_out.read()

        Returns an instance of :class:`~gridfs.grid_file.GridOut`.

        Raises :exc:`~gridfs.errors.NoFile` if no file with file_id exists.

        :Parameters:
          - `file_id`: The _id of the file to be downloaded.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`

        .. versionchanged:: 3.6
           Added ``session`` parameter.
        """
    gout = GridOut(self._collection, file_id, session=session)
    gout._ensure_file()
    return gout