import hashlib
import json
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve
from pyproj._sync import get_proj_endpoint
from pyproj.aoi import BBox
from pyproj.datadir import get_data_dir, get_user_data_dir
def get_transform_grid_list(source_id: Optional[str]=None, area_of_use: Optional[str]=None, filename: Optional[str]=None, bbox: Optional[BBox]=None, spatial_test: str='intersects', include_world_coverage: bool=True, include_already_downloaded: bool=False, target_directory: Optional[str]=None) -> tuple:
    """
    Get a list of transform grids that can be downloaded.

    Parameters
    ----------
    source_id: str, optional
    area_of_use: str, optional
    filename: str, optional
    bbox: BBox, optional
    spatial_test: str, default="intersects"
        Can be "contains" or "intersects".
    include_world_coverage: bool, default=True
        If True, it will include grids with a global extent.
    include_already_downloaded: bool, default=False
        If True, it will list grids regardless of if they are downloaded.
    target_directory: Union[str, Path, None], optional
        The directory to download the geojson file to.
        Default is the user writable directory.

    Returns
    -------
    list[dict[str, Any]]:
        A list of geojson data of containing information about features
        that can be downloaded.
    """
    features = _load_grid_geojson(target_directory=target_directory)['features']
    if bbox is not None:
        if bbox.west > 180 and bbox.east > bbox.west:
            bbox.west -= 360
            bbox.east -= 360
        elif bbox.west < -180 and bbox.east > bbox.west:
            bbox.west += 360
            bbox.east += 360
        elif abs(bbox.west) < 180 and abs(bbox.east) < 180 and (bbox.east < bbox.west):
            bbox.east += 360
        features = filter(partial(_filter_bbox, bbox=bbox, spatial_test=spatial_test, include_world_coverage=include_world_coverage), features)
    features = filter(partial(_filter_properties, source_id=source_id, area_of_use=area_of_use, filename=filename), features)
    if include_already_downloaded:
        return tuple(features)
    return tuple(filter(_filter_download_needed, features))