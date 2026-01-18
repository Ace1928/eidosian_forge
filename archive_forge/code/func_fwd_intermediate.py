import math
import warnings
from typing import Any, Optional, Union
from pyproj._geod import Geod as _Geod
from pyproj._geod import GeodIntermediateReturn, geodesic_version_str
from pyproj._geod import reverse_azimuth as _reverse_azimuth
from pyproj.enums import GeodIntermediateFlag
from pyproj.exceptions import GeodError
from pyproj.list import get_ellps_map
from pyproj.utils import DataType, _convertback, _copytobuffer
def fwd_intermediate(self, lon1: float, lat1: float, azi1: float, npts: int, del_s: float, initial_idx: int=1, terminus_idx: int=1, radians: bool=False, flags: GeodIntermediateFlag=GeodIntermediateFlag.DEFAULT, out_lons: Optional[Any]=None, out_lats: Optional[Any]=None, out_azis: Optional[Any]=None, return_back_azimuth: Optional[bool]=None) -> GeodIntermediateReturn:
    """
        .. versionadded:: 3.1.0
        .. versionadded:: 3.5.0 return_back_azimuth

        Given a single initial point and azimuth, number of points (npts)
        and delimiter distance between two successive points (del_s), returns
        a list of longitude/latitude pairs describing npts equally
        spaced intermediate points along the geodesic between the
        initial and terminus points.

        Example usage:

        >>> from pyproj import Geod
        >>> g = Geod(ellps='clrk66') # Use Clarke 1866 ellipsoid.
        >>> # specify the lat/lons of Boston and Portland.
        >>> boston_lat = 42.+(15./60.); boston_lon = -71.-(7./60.)
        >>> portland_lat = 45.+(31./60.); portland_lon = -123.-(41./60.)
        >>> az12,az21,dist = g.inv(boston_lon,boston_lat,portland_lon,portland_lat)
        >>> # find ten equally spaced points between Boston and Portland.
        >>> npts = 10
        >>> del_s = dist/(npts+1)
        >>> r = g.fwd_intermediate(boston_lon,boston_lat,az12,npts=npts,del_s=del_s)
        >>> for lon,lat in zip(r.lons, r.lats): f'{lat:.3f} {lon:.3f}'
        '43.528 -75.414'
        '44.637 -79.883'
        '45.565 -84.512'
        '46.299 -89.279'
        '46.830 -94.156'
        '47.149 -99.112'
        '47.251 -104.106'
        '47.136 -109.100'
        '46.805 -114.051'
        '46.262 -118.924'
        >>> # test with radians=True (inputs/outputs in radians, not degrees)
        >>> import math
        >>> dg2rad = math.radians(1.)
        >>> rad2dg = math.degrees(1.)
        >>> r = g.fwd_intermediate(
        ...    dg2rad*boston_lon,
        ...    dg2rad*boston_lat,
        ...    dg2rad*az12,
        ...    npts=npts,
        ...    del_s=del_s,
        ...    radians=True
        ... )
        >>> for lon,lat in zip(r.lons, r.lats): f'{rad2dg*lat:.3f} {rad2dg*lon:.3f}'
        '43.528 -75.414'
        '44.637 -79.883'
        '45.565 -84.512'
        '46.299 -89.279'
        '46.830 -94.156'
        '47.149 -99.112'
        '47.251 -104.106'
        '47.136 -109.100'
        '46.805 -114.051'
        '46.262 -118.924'

        Parameters
        ----------
        lon1: float
            Longitude of the initial point
        lat1: float
            Latitude of the initial point
        azi1: float
            Azimuth from the initial point towards the terminus point
        npts: int
            Number of points to be returned
            (including initial and/or terminus points, if required)
        del_s: float
            delimiter distance between two successive points
        radians: bool, default=False
            If True, the input data is assumed to be in radians.
            Otherwise, the data is assumed to be in degrees.
        initial_idx: int, default=1
            if initial_idx==0 then the initial point would be included in the output
            (as the first point)
        terminus_idx: int, default=1
            if terminus_idx==0 then the terminus point would be included in the output
            (as the last point)
        flags: GeodIntermediateFlag, default=GeodIntermediateFlag.DEFAULT
            * 1st - round/ceil/trunc (see ``GeodIntermediateFlag.NPTS_*``)
            * 2nd - update del_s to the new npts or not
                    (see ``GeodIntermediateFlag.DEL_S_*``)
            * 3rd - if out_azis=None, indicates if to save or discard the azimuths
                    (see ``GeodIntermediateFlag.AZIS_*``)
            * default - round npts, update del_s accordingly, discard azis
        out_lons: array, :class:`numpy.ndarray`, optional
            Longitude(s) of the intermediate point(s)
            If None then buffers would be allocated internnaly
        out_lats: array, :class:`numpy.ndarray`, optional
            Latitudes(s) of the intermediate point(s)
            If None then buffers would be allocated internnaly
        out_azis: array, :class:`numpy.ndarray`, optional
            az12(s) of the intermediate point(s)
            If None then buffers would be allocated internnaly
            unless requested otherwise by the flags
        return_back_azimuth: bool, default=True
            if True, out_azis will store the back azimuth,
            Otherwise, out_azis will store the forward azimuth.

        Returns
        -------
        GeodIntermediateReturn:
            number of points, distance and output arrays (GeodIntermediateReturn docs)
        """
    if return_back_azimuth is None:
        return_back_azimuth = True
        warnings.warn('Back azimuth is being returned by default to be compatible with inv()This is a breaking change for pyproj 3.5+.To avoid this warning, set return_back_azimuth=True.Otherwise, to restore old behaviour, set return_back_azimuth=False.This warning will be removed in future version.')
    return super()._inv_or_fwd_intermediate(lon1=lon1, lat1=lat1, lon2_or_azi1=azi1, lat2=math.nan, npts=npts, del_s=del_s, radians=radians, initial_idx=initial_idx, terminus_idx=terminus_idx, flags=int(flags), out_lons=out_lons, out_lats=out_lats, out_azis=out_azis, return_back_azimuth=return_back_azimuth, is_fwd=True)