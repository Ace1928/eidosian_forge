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
def open_upload_stream(self, filename: str, chunk_size_bytes: Optional[int]=None, metadata: Optional[Mapping[str, Any]]=None, session: Optional[ClientSession]=None) -> GridIn:
    """Opens a Stream that the application can write the contents of the
        file to.

        The user must specify the filename, and can choose to add any
        additional information in the metadata field of the file document or
        modify the chunk size.
        For example::

          my_db = MongoClient().test
          fs = GridFSBucket(my_db)
          with fs.open_upload_stream(
                "test_file", chunk_size_bytes=4,
                metadata={"contentType": "text/plain"}) as grid_in:
              grid_in.write("data I want to store!")
          # uploaded on close

        Returns an instance of :class:`~gridfs.grid_file.GridIn`.

        Raises :exc:`~gridfs.errors.NoFile` if no such version of
        that file exists.
        Raises :exc:`~ValueError` if `filename` is not a string.

        :Parameters:
          - `filename`: The name of the file to upload.
          - `chunk_size_bytes` (options): The number of bytes per chunk of this
            file. Defaults to the chunk_size_bytes in :class:`GridFSBucket`.
          - `metadata` (optional): User data for the 'metadata' field of the
            files collection document. If not provided the metadata field will
            be omitted from the files collection document.
          - `session` (optional): a
            :class:`~pymongo.client_session.ClientSession`

        .. versionchanged:: 3.6
           Added ``session`` parameter.
        """
    validate_string('filename', filename)
    opts = {'filename': filename, 'chunk_size': chunk_size_bytes if chunk_size_bytes is not None else self._chunk_size_bytes}
    if metadata is not None:
        opts['metadata'] = metadata
    return GridIn(self._collection, session=session, **opts)