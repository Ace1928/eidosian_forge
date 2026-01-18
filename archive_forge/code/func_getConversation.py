from zope.interface import Attribute, Interface
def getConversation(person, Class, stayHidden=0):
    """
        For the given person object, returns the conversation window
        or creates and returns a new conversation window if one does not exist.

        @type person: L{Person<IPerson>}
        @type Class: L{Conversation<IConversation>} class
        @type stayHidden: boolean

        @rtype: L{Conversation<IConversation>}
        """