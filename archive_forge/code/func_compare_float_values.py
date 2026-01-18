from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def compare_float_values(self, ele1, ele2, precision=2):
    """
        Compare two floating-point values with a specified precision.
        Args:
            - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            - ele1 (float): The first floating-point value to be compared.
            - ele2 (float): The second floating-point value to be compared.
            - precision (int, optional): The number of decimal places to consider in the comparison, Defaults to 2.
        Return:
            bool: True if the rounded values are equal within the specified precision, False otherwise.
        Description:
            This method compares two floating-point values, ele1 and ele2, by rounding them
            to the specified precision and checking if the rounded values are equal. It returns
            True if the rounded values are equal within the specified precision, and False otherwise.
        """
    return round(float(ele1), precision) == round(float(ele2), precision)