from __future__ import annotations
from abc import abstractmethod
from enum import Enum
from typing import Protocol
@abstractmethod
def load_and_get_id(self, path_or_data: str | bytes, mimetype: str, kind: MediaFileKind, filename: str | None=None) -> str:
    """Load the given file path or bytes into the manager and return
        an ID that uniquely identifies it.

        It's an error to pass a URL to this function. (Media stored at
        external URLs can be served directly to the Streamlit frontend;
        there's no need to store this data in MediaFileStorage.)

        Parameters
        ----------
        path_or_data
            A path to a file, or the file's raw data as bytes.

        mimetype
            The media's mimetype. Used to set the Content-Type header when
            serving the media over HTTP.

        kind
            The kind of file this is: either MEDIA, or DOWNLOADABLE.

        filename : str or None
            Optional filename. Used to set the filename in the response header.

        Returns
        -------
        str
            The unique ID of the media file.

        Raises
        ------
        MediaFileStorageError
            Raised if the media can't be loaded (for example, if a file
            path is invalid).

        """
    raise NotImplementedError