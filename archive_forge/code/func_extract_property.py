from abc import ABCMeta, abstractmethod
@staticmethod
@abstractmethod
def extract_property(tokens, index):
    """
        Any subclass of Feature must define static method extract_property(tokens, index)

        :param tokens: the sequence of tokens
        :type tokens: list of tokens
        :param index: the current index
        :type index: int
        :return: feature value
        :rtype: any (but usually scalar)
        """