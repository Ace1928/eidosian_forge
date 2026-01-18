from prov.model import ProvDocument
def document_with_n_bundles_having_default_namespace(n):
    prov_doc = ProvDocument()
    prov_doc.add_namespace('ex', 'http://www.example.org/')
    for i in range(n):
        x = str(i + 1)
        bundle = prov_doc.bundle('ex:bundle/' + x)
        bundle.set_default_namespace('http://www.example.org/default/' + x)
        bundle.entity('e')
    return prov_doc