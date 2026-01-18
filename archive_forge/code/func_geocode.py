from collections import defaultdict
import time
import pandas as pd
from shapely.geometry import Point
import geopandas
def geocode(strings, provider=None, **kwargs):
    """
    Geocode a set of strings and get a GeoDataFrame of the resulting points.

    Parameters
    ----------
    strings : list or Series of addresses to geocode
    provider : str or geopy.geocoder
        Specifies geocoding service to use. If none is provided,
        will use 'photon' (see the Photon's terms of service at:
        https://photon.komoot.io).

        Either the string name used by geopy (as specified in
        geopy.geocoders.SERVICE_TO_GEOCODER) or a geopy Geocoder instance
        (e.g., geopy.geocoders.Photon) may be used.

        Some providers require additional arguments such as access keys
        See each geocoder's specific parameters in geopy.geocoders

    Notes
    -----
    Ensure proper use of the results by consulting the Terms of Service for
    your provider.

    Geocoding requires geopy. Install it using 'pip install geopy'. See also
    https://github.com/geopy/geopy

    Examples
    --------
    >>> df = geopandas.tools.geocode(  # doctest: +SKIP
    ...         ["boston, ma", "1600 pennsylvania ave. washington, dc"]
    ...     )
    >>> df  # doctest: +SKIP
                        geometry                                            address
    0  POINT (-71.05863 42.35899)                          Boston, MA, United States
    1  POINT (-77.03651 38.89766)  1600 Pennsylvania Ave NW, Washington, DC 20006...
    """
    if provider is None:
        provider = 'photon'
    throttle_time = _get_throttle_time(provider)
    return _query(strings, True, provider, throttle_time, **kwargs)