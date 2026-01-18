import boto
import boto.jsonresponse
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def define_index_field(self, domain_name, field_name, field_type, default='', facet=False, result=False, searchable=False, source_attributes=None):
    """
        Defines an ``IndexField``, either replacing an existing
        definition or creating a new one.

        :type domain_name: string
        :param domain_name: A string that represents the name of a
            domain. Domain names must be unique across the domains
            owned by an account within an AWS region. Domain names
            must start with a letter or number and can contain the
            following characters: a-z (lowercase), 0-9, and -
            (hyphen). Uppercase letters and underscores are not
            allowed.

        :type field_name: string
        :param field_name: The name of a field in the search index.

        :type field_type: string
        :param field_type: The type of field.  Valid values are
            uint | literal | text

        :type default: string or int
        :param default: The default value for the field.  If the
            field is of type ``uint`` this should be an integer value.
            Otherwise, it's a string.

        :type facet: bool
        :param facet: A boolean to indicate whether facets
            are enabled for this field or not.  Does not apply to
            fields of type ``uint``.

        :type results: bool
        :param results: A boolean to indicate whether values
            of this field can be returned in search results or
            used in ranking.  Does not apply to fields of type ``uint``.

        :type searchable: bool
        :param searchable: A boolean to indicate whether search
            is enabled for this field or not.  Applies only to fields
            of type ``literal``.

        :type source_attributes: list of dicts
        :param source_attributes: An optional list of dicts that
            provide information about attributes for this index field.
            A maximum of 20 source attributes can be configured for
            each index field.

            Each item in the list is a dict with the following keys:

            * data_copy - The value is a dict with the following keys:
                * default - Optional default value if the source attribute
                    is not specified in a document.
                * name - The name of the document source field to add
                    to this ``IndexField``.
            * data_function - Identifies the transformation to apply
                when copying data from a source attribute.
            * data_map - The value is a dict with the following keys:
                * cases - A dict that translates source field values
                    to custom values.
                * default - An optional default value to use if the
                    source attribute is not specified in a document.
                * name - the name of the document source field to add
                    to this ``IndexField``
            * data_trim_title - Trims common title words from a source
                document attribute when populating an ``IndexField``.
                This can be used to create an ``IndexField`` you can
                use for sorting.  The value is a dict with the following
                fields:
                * default - An optional default value.
                * language - an IETF RFC 4646 language code.
                * separator - The separator that follows the text to trim.
                * name - The name of the document source field to add.

        :raises: BaseException, InternalException, LimitExceededException,
            InvalidTypeException, ResourceNotFoundException
        """
    doc_path = ('define_index_field_response', 'define_index_field_result', 'index_field')
    params = {'DomainName': domain_name, 'IndexField.IndexFieldName': field_name, 'IndexField.IndexFieldType': field_type}
    if field_type == 'literal':
        params['IndexField.LiteralOptions.DefaultValue'] = default
        params['IndexField.LiteralOptions.FacetEnabled'] = do_bool(facet)
        params['IndexField.LiteralOptions.ResultEnabled'] = do_bool(result)
        params['IndexField.LiteralOptions.SearchEnabled'] = do_bool(searchable)
    elif field_type == 'uint':
        params['IndexField.UIntOptions.DefaultValue'] = default
    elif field_type == 'text':
        params['IndexField.TextOptions.DefaultValue'] = default
        params['IndexField.TextOptions.FacetEnabled'] = do_bool(facet)
        params['IndexField.TextOptions.ResultEnabled'] = do_bool(result)
    return self.get_response(doc_path, 'DefineIndexField', params, verb='POST')