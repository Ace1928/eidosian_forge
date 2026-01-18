import os
import re
import configparser
import keystone.conf
def symptom_LDAP_file_based_domain_specific_configs():
    """Domain specific driver directory is invalid or contains invalid files.

    If `keystone.conf [identity] domain_specific_drivers_enabled` is set
    to `true`, then support is enabled for individual domains to have their
    own identity drivers. The configurations for these can either be stored
    in a config file or in the database. The case we handle in this symptom
    is when they are stored in config files, which is indicated by
    `keystone.conf [identity] domain_configurations_from_database`
    being set to `false`.
    """
    if not CONF.identity.domain_specific_drivers_enabled or CONF.identity.domain_configurations_from_database:
        return False
    invalid_files = []
    filedir = CONF.identity.domain_config_dir
    if os.path.isdir(filedir):
        for filename in os.listdir(filedir):
            if not re.match(CONFIG_REGEX, filename):
                invalid_files.append(filename)
        if invalid_files:
            invalid_str = ', '.join(invalid_files)
            print('Warning: The following non-config files were found: %s\nIf they are intended to be config files then rename them to the form of `keystone.<domain_name>.conf`. Otherwise, ignore this warning' % invalid_str)
            return True
    else:
        print('Could not find directory ', filedir)
        return True
    return False