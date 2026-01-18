from pbr.hooks import backwards
from pbr.hooks import commands
from pbr.hooks import files
from pbr.hooks import metadata
def setup_hook(config):
    """Filter config parsed from a setup.cfg to inject our defaults."""
    metadata_config = metadata.MetadataConfig(config)
    metadata_config.run()
    backwards.BackwardsCompatConfig(config).run()
    commands.CommandsConfig(config).run()
    files.FilesConfig(config, metadata_config.get_name()).run()