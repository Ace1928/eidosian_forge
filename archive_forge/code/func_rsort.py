import logging
import re
def rsort(list_, loose):
    keyf = loose_key_function if loose else full_key_function
    list_.sort(key=keyf, reverse=True)
    return list_