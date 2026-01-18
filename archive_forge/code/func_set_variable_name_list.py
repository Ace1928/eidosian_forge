import itertools
def set_variable_name_list(self, variable_name_list):
    """
        Specify variable names with its full name.

        Parameters
        ----------
        variable_name_list: a ``list`` of ``string``, containing the variable names with indices,
            for e.g. "C['CA', 23, 0]".
        """
    super().set_variable_name_list(variable_name_list)