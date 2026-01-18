from __future__ import annotations
import json
import urllib.request
import uuid
from typing import Callable
from urllib.parse import quote
def query_name(self, name: str) -> TileProvider:
    """Return :class:`TileProvider` based on the name query

        Returns a matching :class:`TileProvider` from the :class:`Bunch` if the ``name``
        contains the same letters in the same order as the provider's name irrespective
        of the letter case, spaces, dashes and other characters.
        See examples for details.

        Parameters
        ----------
        name : str
            Name of the tile provider. Formatting does not matter.

        Returns
        -------
        match: TileProvider

        Examples
        --------
        >>> import xyzservices.providers as xyz

        All these queries return the same ``CartoDB.Positron`` TileProvider:

        >>> xyz.query_name("CartoDB Positron")
        >>> xyz.query_name("cartodbpositron")
        >>> xyz.query_name("cartodb-positron")
        >>> xyz.query_name("carto db/positron")
        >>> xyz.query_name("CARTO_DB_POSITRON")
        >>> xyz.query_name("CartoDB.Positron")

        """
    xyz_flat_lower = {k.translate(QUERY_NAME_TRANSLATION).lower(): v for k, v in self.flatten().items()}
    name_clean = name.translate(QUERY_NAME_TRANSLATION).lower()
    if name_clean in xyz_flat_lower:
        return xyz_flat_lower[name_clean]
    raise ValueError(f"No matching provider found for the query '{name}'.")