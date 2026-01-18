from __future__ import (absolute_import, division, print_function)
@staticmethod
def paramgram_child_list_override(list_overrides, paramgram, module):
    """
        If a list of items was provided to a "parent" paramgram attribute, the paramgram needs to be rewritten.
        The child keys of the desired attribute need to be deleted, and then that "parent" keys' contents is replaced
        With the list of items that was provided.

        :param list_overrides: Contains the response from the FortiManager.
        :type list_overrides: list
        :param paramgram: Contains the paramgram passed to the modules' local modify function.
        :type paramgram: dict
        :param module: Contains the Ansible Module Object being used by the module.
        :type module: classObject

        :return: A new "paramgram" refactored to allow for multiple entries being added.
        :rtype: dict
        """
    if len(list_overrides) > 0:
        for list_variable in list_overrides:
            try:
                list_variable = list_variable.replace('-', '_')
                override_data = module.params[list_variable]
                if override_data:
                    del paramgram[list_variable]
                    paramgram[list_variable] = override_data
            except BaseException as e:
                raise FMGBaseException('Error occurred merging custom lists for the paramgram parent: ' + str(e))
    return paramgram