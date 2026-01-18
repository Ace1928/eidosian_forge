def election_geojson():
    """
    Each feature represents an electoral district in the 2013 Montreal mayoral election.

    Returns:
        A GeoJSON-formatted `dict` with 58 polygon or multi-polygon features whose `id`
        is an electoral district numerical ID and whose `district` property is the ID and
        district name."""
    import gzip
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'package_data', 'datasets', 'election.geojson.gz')
    with gzip.GzipFile(path, 'r') as f:
        result = json.loads(f.read().decode('utf-8'))
    return result