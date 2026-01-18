import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
def add_volume(volume, volumes, messages, release_track):
    """Add the volume described by the given volume dict to the resource."""
    if 'name' not in volume or 'type' not in volume:
        raise serverless_exceptions.ConfigurationError('All added volumes must have a name and type')
    if volume['type'] not in _supported_volume_types:
        raise serverless_exceptions.ConfigurationError('Volume type {} not supported'.format(volume['type']))
    new_vol = messages.Volume(name=volume['name'])
    vol_type = _supported_volume_types[volume['type']]
    if release_track not in vol_type.release_tracks():
        raise serverless_exceptions.ConfigurationError('Volume type {} not supported'.format(volume['type']))
    vol_type.validate_volume_add(volume)
    vol_type.fill_volume(volume, new_vol, messages)
    volumes[volume['name']] = new_vol