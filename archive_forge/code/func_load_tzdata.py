import struct
def load_tzdata(key):
    from importlib import resources
    components = key.split('/')
    package_name = '.'.join(['tzdata.zoneinfo'] + components[:-1])
    resource_name = components[-1]
    try:
        return resources.files(package_name).joinpath(resource_name).open('rb')
    except (ImportError, FileNotFoundError, UnicodeEncodeError):
        raise ZoneInfoNotFoundError(f'No time zone found with key {key}')