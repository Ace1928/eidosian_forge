import abc
@abc.abstractmethod
def start_protocol(self, socket):
    """Launch protocol instance to handle input on an incoming connection.
        """
    raise NotImplementedError()