import configparser
import copy
import os
import shlex
import sys
import botocore.exceptions
def raw_config_parse(config_filename, parse_subsections=True):
    """Returns the parsed INI config contents.

    Each section name is a top level key.

    :param config_filename: The name of the INI file to parse

    :param parse_subsections: If True, parse indented blocks as
       subsections that represent their own configuration dictionary.
       For example, if the config file had the contents::

           s3 =
              signature_version = s3v4
              addressing_style = path

        The resulting ``raw_config_parse`` would be::

            {'s3': {'signature_version': 's3v4', 'addressing_style': 'path'}}

       If False, do not try to parse subsections and return the indented
       block as its literal value::

            {'s3': '
signature_version = s3v4
addressing_style = path'}

    :returns: A dict with keys for each profile found in the config
        file and the value of each key being a dict containing name
        value pairs found in that profile.

    :raises: ConfigNotFound, ConfigParseError
    """
    config = {}
    path = config_filename
    if path is not None:
        path = os.path.expandvars(path)
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise botocore.exceptions.ConfigNotFound(path=_unicode_path(path))
        cp = configparser.RawConfigParser()
        try:
            cp.read([path])
        except (configparser.Error, UnicodeDecodeError) as e:
            raise botocore.exceptions.ConfigParseError(path=_unicode_path(path), error=e) from None
        else:
            for section in cp.sections():
                config[section] = {}
                for option in cp.options(section):
                    config_value = cp.get(section, option)
                    if parse_subsections and config_value.startswith('\n'):
                        try:
                            config_value = _parse_nested(config_value)
                        except ValueError as e:
                            raise botocore.exceptions.ConfigParseError(path=_unicode_path(path), error=e) from None
                    config[section][option] = config_value
    return config