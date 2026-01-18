def get_violating_attr(self, attr_name):
    """
        If separation problems solved globally, returns
        value of attribute of global separation loop results.

        Otherwise, if separation problems solved locally,
        returns value of attribute of local separation loop results.
        If local separation loop results specified, return
        value of attribute of local separation loop results.

        Otherwise, if global separation loop results specified,
        return value of attribute of global separation loop
        results.

        Otherwise, return None.

        Parameters
        ----------
        attr_name : str
            Name of attribute to be retrieved. Should be
            valid attribute name for object of type
            ``SeparationLoopResults``.

        Returns
        -------
        object
            Attribute value.
        """
    return getattr(self.main_loop_results, attr_name, None)