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
@_csot.apply
def download_to_stream_by_name(self, filename: str, destination: Any, revision: int=-1, session: Optional[ClientSession]=None) -> None:
    """Write the contents of `filename` (with optional `revision`) to
        `destination`.

        For example::

          my_db = MongoClient().test
          fs = GridFSBucket(my_db)
          # Get file to write to
          file = open('myfile','wb')
          fs.download_to_stream_by_name("test_file", file)

        Raises :exc:`~gridfs.errors.NoFile` if no such version of
        that file exists.

        Raises :exc:`~ValueError` if `filename` is not a string.

        :Parameters:
          - `filename`: The name of the file to read from.
          - `destination`: A file-like object that implements :meth:`write`.
          - `revision` (optional): Which revision (documents with the same
            filename and different uploadDate) of the file to retrieve.
            Defaults to -1 (the most recent revision).
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`

        :Note: Revision numbers are defined as follows:

          - 0 = the original stored file
          - 1 = the first revision
          - 2 = the second revision
          - etc...
          - -2 = the second most recent revision
          - -1 = the most recent revision

        .. versionchanged:: 3.6
           Added ``session`` parameter.
        """
    with self.open_download_stream_by_name(filename, revision, session=session) as gout:
        while True:
            chunk = gout.readchunk()
            if not len(chunk):
                break
            destination.write(chunk)