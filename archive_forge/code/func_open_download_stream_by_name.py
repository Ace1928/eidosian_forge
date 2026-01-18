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
def open_download_stream_by_name(self, filename: str, revision: int=-1, session: Optional[ClientSession]=None) -> GridOut:
    """Opens a Stream from which the application can read the contents of
        `filename` and optional `revision`.

        For example::

          my_db = MongoClient().test
          fs = GridFSBucket(my_db)
          grid_out = fs.open_download_stream_by_name("test_file")
          contents = grid_out.read()

        Returns an instance of :class:`~gridfs.grid_file.GridOut`.

        Raises :exc:`~gridfs.errors.NoFile` if no such version of
        that file exists.

        Raises :exc:`~ValueError` filename is not a string.

        :Parameters:
          - `filename`: The name of the file to read from.
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
    validate_string('filename', filename)
    query = {'filename': filename}
    _disallow_transactions(session)
    cursor = self._files.find(query, session=session)
    if revision < 0:
        skip = abs(revision) - 1
        cursor.limit(-1).skip(skip).sort('uploadDate', DESCENDING)
    else:
        cursor.limit(-1).skip(revision).sort('uploadDate', ASCENDING)
    try:
        grid_file = next(cursor)
        return GridOut(self._collection, file_document=grid_file, session=session)
    except StopIteration:
        raise NoFile('no version %d for filename %r' % (revision, filename)) from None