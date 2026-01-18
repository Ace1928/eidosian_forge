from pyparsing import *
import random
import string
def makeBNF(self):
    invVerb = oneOf('INV INVENTORY I', caseless=True)
    dropVerb = oneOf('DROP LEAVE', caseless=True)
    takeVerb = oneOf('TAKE PICKUP', caseless=True) | CaselessLiteral('PICK') + CaselessLiteral('UP')
    moveVerb = oneOf('MOVE GO', caseless=True) | empty
    useVerb = oneOf('USE U', caseless=True)
    openVerb = oneOf('OPEN O', caseless=True)
    closeVerb = oneOf('CLOSE CL', caseless=True)
    quitVerb = oneOf('QUIT Q', caseless=True)
    lookVerb = oneOf('LOOK L', caseless=True)
    doorsVerb = CaselessLiteral('DOORS')
    helpVerb = oneOf('H HELP ?', caseless=True)
    itemRef = OneOrMore(Word(alphas)).setParseAction(self.validateItemName)
    nDir = oneOf('N NORTH', caseless=True).setParseAction(replaceWith('N'))
    sDir = oneOf('S SOUTH', caseless=True).setParseAction(replaceWith('S'))
    eDir = oneOf('E EAST', caseless=True).setParseAction(replaceWith('E'))
    wDir = oneOf('W WEST', caseless=True).setParseAction(replaceWith('W'))
    moveDirection = nDir | sDir | eDir | wDir
    invCommand = invVerb
    dropCommand = dropVerb + itemRef('item')
    takeCommand = takeVerb + itemRef('item')
    useCommand = useVerb + itemRef('usedObj') + Optional(oneOf('IN ON', caseless=True)) + Optional(itemRef, default=None)('targetObj')
    openCommand = openVerb + itemRef('item')
    closeCommand = closeVerb + itemRef('item')
    moveCommand = moveVerb + moveDirection('direction')
    quitCommand = quitVerb
    lookCommand = lookVerb
    doorsCommand = doorsVerb
    helpCommand = helpVerb
    invCommand.setParseAction(InventoryCommand)
    dropCommand.setParseAction(DropCommand)
    takeCommand.setParseAction(TakeCommand)
    useCommand.setParseAction(UseCommand)
    openCommand.setParseAction(OpenCommand)
    closeCommand.setParseAction(CloseCommand)
    moveCommand.setParseAction(MoveCommand)
    quitCommand.setParseAction(QuitCommand)
    lookCommand.setParseAction(LookCommand)
    doorsCommand.setParseAction(DoorsCommand)
    helpCommand.setParseAction(HelpCommand)
    return (invCommand | useCommand | openCommand | closeCommand | dropCommand | takeCommand | moveCommand | lookCommand | doorsCommand | helpCommand | quitCommand)('command') + LineEnd()