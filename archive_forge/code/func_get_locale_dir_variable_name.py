def get_locale_dir_variable_name(domain):
    """Build environment variable name for  local dir.

    Convert a translation domain name to a variable for specifying
    a separate locale dir.

    """
    return domain.upper().replace('.', '_').replace('-', '_') + '_LOCALEDIR'