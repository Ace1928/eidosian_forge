from zope.interface import Attribute, Interface
def contactChangedNick(oldnick, newnick):
    """
        For the given person, changes the person's name to newnick, and
        tells the contact list and any conversation windows with that person
        to change as well.

        @type oldnick: string
        @type newnick: string
        """