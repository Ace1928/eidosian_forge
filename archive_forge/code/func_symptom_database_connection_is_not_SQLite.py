import keystone.conf
def symptom_database_connection_is_not_SQLite():
    """SQLite is not recommended for production deployments.

    SQLite does not enforce type checking and has limited support for
    migrations, making it unsuitable for use in keystone. Please change your
    `keystone.conf [database] connection` value to point to a supported
    database driver, such as MySQL.
    """
    return CONF.database.connection is not None and 'sqlite' in CONF.database.connection