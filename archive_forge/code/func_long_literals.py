from prov.model import ProvDocument, Namespace, Literal, PROV, Identifier
import datetime
def long_literals():
    g = ProvDocument()
    long_uri = 'http://Lorem.ipsum/dolor/sit/amet/consectetur/adipiscing/elit/Quisque/vel/sollicitudin/felis/nec/venenatis/massa/Aenean/lectus/arcu/sagittis/sit/amet/nisl/nec/varius/eleifend/sem/In/hac/habitasse/platea/dictumst/Aliquam/eget/fermentum/enim/Curabitur/auctor/elit/non/ipsum/interdum/at/orci/aliquam/'
    ex = Namespace('ex', long_uri)
    g.add_namespace(ex)
    g.entity('ex:e1', {'prov:label': 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec pellentesque luctus nulla vel ullamcorper. Donec sit amet ligula sit amet lorem pretium rhoncus vel vel lorem. Sed at consequat metus, eget eleifend massa. Fusce a facilisis turpis. Lorem volutpat.'})
    return g