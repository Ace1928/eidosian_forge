def superior(self):
    """Returns another instance of language_tag which is the superior.

        Thus en-US gives en, and en gives *.

        """
    if len(self) <= 1:
        return self.__class__('*')
    return self.__class__('-'.join(self.parts[:-1]))