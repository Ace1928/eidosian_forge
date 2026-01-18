def records_from_geo_interface(data):
    """Un-nest data from object implementing __geo_interface__ standard"""
    flattened_records = []
    for d in data.__geo_interface__.get('features'):
        record = d.get('properties', {})
        geom = d.get('geometry', {})
        record['geometry'] = geom
        flattened_records.append(record)
    return flattened_records