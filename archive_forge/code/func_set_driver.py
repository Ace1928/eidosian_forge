def set_driver(drivers, provider, module, klass):
    """
    Sets a driver.

    :param drivers: Dictionary to store providers.
    :param provider: Id of provider to set driver for

    :type provider: :class:`libcloud.types.Provider`
    :param module: The module which contains the driver

    :type module: L
    :param klass: The driver class name

    :type klass:
    """
    if provider in drivers:
        raise AttributeError('Provider %s already registered' % provider)
    drivers[provider] = (module, klass)
    try:
        driver = get_driver(drivers, provider)
    except (ImportError, AttributeError) as exp:
        drivers.pop(provider)
        raise exp
    return driver